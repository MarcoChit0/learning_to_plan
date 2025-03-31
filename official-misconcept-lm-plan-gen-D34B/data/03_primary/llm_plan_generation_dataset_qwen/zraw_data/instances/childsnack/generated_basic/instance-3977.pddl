; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.9
; random seed: 416251

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child81 - child
    bread424 - bread-portion
    content302 - content-portion
    tray95 tray237 tray398 - tray
    table201 table300 table498 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray95 kitchen)
     (at tray237 kitchen)
     (at tray398 kitchen)
     (at_kitchen_bread bread424)
     (at_kitchen_content content302)
     (not_allergic_gluten child81)
     (waiting child81 table498)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child81)
    )
  )
)
