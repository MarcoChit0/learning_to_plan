; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 420723

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child345 child254 - child
    bread430 bread3 - bread-portion
    content5 content145 - content-portion
    tray424 - tray
    table36 table441 table491 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray424 kitchen)
     (at_kitchen_bread bread430)
     (at_kitchen_bread bread3)
     (at_kitchen_content content5)
     (at_kitchen_content content145)
     (no_gluten_bread bread430)
     (no_gluten_content content5)
     (allergic_gluten child254)
     (not_allergic_gluten child345)
     (waiting child345 table36)
     (waiting child254 table491)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child345)
     (served child254)
    )
  )
)
