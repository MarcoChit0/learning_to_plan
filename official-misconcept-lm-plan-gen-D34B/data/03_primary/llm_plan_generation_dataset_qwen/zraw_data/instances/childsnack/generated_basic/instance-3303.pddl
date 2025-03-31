; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 241193

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child317 child376 - child
    bread148 bread219 - bread-portion
    content202 content496 - content-portion
    tray330 tray151 tray250 tray49 - tray
    table28 table412 table384 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray330 kitchen)
     (at tray151 kitchen)
     (at tray250 kitchen)
     (at tray49 kitchen)
     (at_kitchen_bread bread148)
     (at_kitchen_bread bread219)
     (at_kitchen_content content202)
     (at_kitchen_content content496)
     (no_gluten_bread bread219)
     (no_gluten_content content496)
     (allergic_gluten child317)
     (not_allergic_gluten child376)
     (waiting child317 table28)
     (waiting child376 table384)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child317)
     (served child376)
    )
  )
)
