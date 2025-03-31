; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 621441

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child1 child115 - child
    bread217 bread336 - bread-portion
    content231 content489 - content-portion
    tray97 tray85 tray203 - tray
    table437 table409 table383 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray97 kitchen)
     (at tray85 kitchen)
     (at tray203 kitchen)
     (at_kitchen_bread bread217)
     (at_kitchen_bread bread336)
     (at_kitchen_content content231)
     (at_kitchen_content content489)
     (not_allergic_gluten child1)
     (not_allergic_gluten child115)
     (waiting child1 table437)
     (waiting child115 table437)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child1)
     (served child115)
    )
  )
)
