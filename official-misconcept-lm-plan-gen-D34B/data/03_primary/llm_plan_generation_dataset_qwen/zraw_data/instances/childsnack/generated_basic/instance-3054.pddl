; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 949902

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child26 child455 - child
    bread183 bread341 - bread-portion
    content399 content365 - content-portion
    tray394 tray429 tray325 - tray
    table15 table403 table416 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray394 kitchen)
     (at tray429 kitchen)
     (at tray325 kitchen)
     (at_kitchen_bread bread183)
     (at_kitchen_bread bread341)
     (at_kitchen_content content399)
     (at_kitchen_content content365)
     (no_gluten_bread bread341)
     (no_gluten_content content399)
     (allergic_gluten child26)
     (not_allergic_gluten child455)
     (waiting child26 table416)
     (waiting child455 table403)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child26)
     (served child455)
    )
  )
)
