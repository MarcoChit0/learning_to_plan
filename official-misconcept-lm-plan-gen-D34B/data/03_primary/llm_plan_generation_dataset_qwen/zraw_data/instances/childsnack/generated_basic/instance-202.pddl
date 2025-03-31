; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 486139

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child421 child440 - child
    bread355 bread446 - bread-portion
    content174 content483 - content-portion
    tray479 tray387 tray486 - tray
    table429 table12 table341 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray479 kitchen)
     (at tray387 kitchen)
     (at tray486 kitchen)
     (at_kitchen_bread bread355)
     (at_kitchen_bread bread446)
     (at_kitchen_content content174)
     (at_kitchen_content content483)
     (not_allergic_gluten child421)
     (not_allergic_gluten child440)
     (waiting child421 table12)
     (waiting child440 table429)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child421)
     (served child440)
    )
  )
)
