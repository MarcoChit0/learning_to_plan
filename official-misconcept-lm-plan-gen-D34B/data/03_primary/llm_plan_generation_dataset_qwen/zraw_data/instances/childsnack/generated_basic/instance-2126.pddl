; child-snack task with 2 children and 0.53 gluten factor 
; constant factor of 1.9
; random seed: 380311

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child55 child407 - child
    bread463 bread88 - bread-portion
    content266 content479 - content-portion
    tray426 tray86 tray409 - tray
    table196 table379 table166 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray426 kitchen)
     (at tray86 kitchen)
     (at tray409 kitchen)
     (at_kitchen_bread bread463)
     (at_kitchen_bread bread88)
     (at_kitchen_content content266)
     (at_kitchen_content content479)
     (no_gluten_bread bread88)
     (no_gluten_content content479)
     (allergic_gluten child407)
     (not_allergic_gluten child55)
     (waiting child55 table379)
     (waiting child407 table196)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child55)
     (served child407)
    )
  )
)
