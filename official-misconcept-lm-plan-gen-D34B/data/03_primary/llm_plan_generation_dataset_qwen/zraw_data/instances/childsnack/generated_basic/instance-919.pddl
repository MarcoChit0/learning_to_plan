; child-snack task with 3 children and 0.31 gluten factor 
; constant factor of 1.5
; random seed: 629482

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child82 child61 child355 - child
    bread115 bread126 bread246 - bread-portion
    content59 content248 content121 - content-portion
    tray2 - tray
    table225 table310 table434 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 - sandwich
  )
  (:init
     (at tray2 kitchen)
     (at_kitchen_bread bread115)
     (at_kitchen_bread bread126)
     (at_kitchen_bread bread246)
     (at_kitchen_content content59)
     (at_kitchen_content content248)
     (at_kitchen_content content121)
     (not_allergic_gluten child82)
     (not_allergic_gluten child355)
     (not_allergic_gluten child61)
     (waiting child82 table434)
     (waiting child61 table225)
     (waiting child355 table434)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
  )
  (:goal
    (and
     (served child82)
     (served child61)
     (served child355)
    )
  )
)
