; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 88206

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child282 - child
    bread236 - bread-portion
    content483 - content-portion
    tray309 - tray
    table272 table311 table7 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray309 kitchen)
     (at_kitchen_bread bread236)
     (at_kitchen_content content483)
     (not_allergic_gluten child282)
     (waiting child282 table311)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child282)
    )
  )
)
