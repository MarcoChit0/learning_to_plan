; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 585546

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child464 child186 - child
    bread244 bread377 - bread-portion
    content239 content348 - content-portion
    tray420 tray42 tray233 tray156 - tray
    table70 table439 table204 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray420 kitchen)
     (at tray42 kitchen)
     (at tray233 kitchen)
     (at tray156 kitchen)
     (at_kitchen_bread bread244)
     (at_kitchen_bread bread377)
     (at_kitchen_content content239)
     (at_kitchen_content content348)
     (no_gluten_bread bread244)
     (no_gluten_content content239)
     (allergic_gluten child186)
     (not_allergic_gluten child464)
     (waiting child464 table70)
     (waiting child186 table204)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child464)
     (served child186)
    )
  )
)
