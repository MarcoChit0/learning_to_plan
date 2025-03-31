; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 719787

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child46 child143 - child
    bread290 bread99 - bread-portion
    content399 content465 - content-portion
    tray317 tray251 tray265 - tray
    table84 table139 table28 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray317 kitchen)
     (at tray251 kitchen)
     (at tray265 kitchen)
     (at_kitchen_bread bread290)
     (at_kitchen_bread bread99)
     (at_kitchen_content content399)
     (at_kitchen_content content465)
     (no_gluten_bread bread99)
     (no_gluten_content content465)
     (allergic_gluten child46)
     (not_allergic_gluten child143)
     (waiting child46 table139)
     (waiting child143 table139)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child46)
     (served child143)
    )
  )
)
