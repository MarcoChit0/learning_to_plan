; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.3
; random seed: 19892

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child181 child24 - child
    bread49 bread120 - bread-portion
    content185 content130 - content-portion
    tray188 tray100 - tray
    table180 table450 table58 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray188 kitchen)
     (at tray100 kitchen)
     (at_kitchen_bread bread49)
     (at_kitchen_bread bread120)
     (at_kitchen_content content185)
     (at_kitchen_content content130)
     (no_gluten_bread bread49)
     (no_gluten_content content185)
     (allergic_gluten child24)
     (not_allergic_gluten child181)
     (waiting child181 table58)
     (waiting child24 table180)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child181)
     (served child24)
    )
  )
)
