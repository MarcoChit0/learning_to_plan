; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 920263

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child227 - child
    bread449 - bread-portion
    content399 - content-portion
    tray429 tray294 tray422 tray433 - tray
    table98 table300 table370 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray429 kitchen)
     (at tray294 kitchen)
     (at tray422 kitchen)
     (at tray433 kitchen)
     (at_kitchen_bread bread449)
     (at_kitchen_content content399)
     (not_allergic_gluten child227)
     (waiting child227 table98)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child227)
    )
  )
)
