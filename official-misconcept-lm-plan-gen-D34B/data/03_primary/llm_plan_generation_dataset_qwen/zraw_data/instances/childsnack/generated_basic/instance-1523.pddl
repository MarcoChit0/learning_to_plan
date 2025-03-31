; child-snack task with 2 children and 0.66 gluten factor 
; constant factor of 1.3
; random seed: 122535

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child54 child58 - child
    bread360 bread451 - bread-portion
    content277 content337 - content-portion
    tray48 tray333 tray496 - tray
    table147 table411 table62 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray48 kitchen)
     (at tray333 kitchen)
     (at tray496 kitchen)
     (at_kitchen_bread bread360)
     (at_kitchen_bread bread451)
     (at_kitchen_content content277)
     (at_kitchen_content content337)
     (no_gluten_bread bread360)
     (no_gluten_content content277)
     (allergic_gluten child58)
     (not_allergic_gluten child54)
     (waiting child54 table411)
     (waiting child58 table411)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child54)
     (served child58)
    )
  )
)
