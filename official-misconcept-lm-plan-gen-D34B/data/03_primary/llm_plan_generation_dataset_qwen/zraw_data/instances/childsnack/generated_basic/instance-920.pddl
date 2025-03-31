; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 784549

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child133 child331 child492 - child
    bread321 bread305 bread279 - bread-portion
    content33 content431 content492 - content-portion
    tray14 - tray
    table339 table489 table63 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray14 kitchen)
     (at_kitchen_bread bread321)
     (at_kitchen_bread bread305)
     (at_kitchen_bread bread279)
     (at_kitchen_content content33)
     (at_kitchen_content content431)
     (at_kitchen_content content492)
     (not_allergic_gluten child492)
     (not_allergic_gluten child331)
     (not_allergic_gluten child133)
     (waiting child133 table63)
     (waiting child331 table489)
     (waiting child492 table489)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child133)
     (served child331)
     (served child492)
    )
  )
)
