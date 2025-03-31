; child-snack task with 1 children and 0.43 gluten factor 
; constant factor of 1.9
; random seed: 692847

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child180 - child
    bread130 - bread-portion
    content45 - content-portion
    tray40 tray199 tray271 tray237 - tray
    table224 table72 table475 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray40 kitchen)
     (at tray199 kitchen)
     (at tray271 kitchen)
     (at tray237 kitchen)
     (at_kitchen_bread bread130)
     (at_kitchen_content content45)
     (not_allergic_gluten child180)
     (waiting child180 table72)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child180)
    )
  )
)
