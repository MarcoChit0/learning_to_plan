; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.9
; random seed: 480048

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child416 child180 - child
    bread379 bread45 - bread-portion
    content95 content71 - content-portion
    tray117 tray192 tray114 tray273 - tray
    table448 table36 table430 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray117 kitchen)
     (at tray192 kitchen)
     (at tray114 kitchen)
     (at tray273 kitchen)
     (at_kitchen_bread bread379)
     (at_kitchen_bread bread45)
     (at_kitchen_content content95)
     (at_kitchen_content content71)
     (not_allergic_gluten child416)
     (not_allergic_gluten child180)
     (waiting child416 table36)
     (waiting child180 table430)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child416)
     (served child180)
    )
  )
)
