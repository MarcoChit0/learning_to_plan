; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.9
; random seed: 170070

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child254 child409 - child
    bread122 bread237 - bread-portion
    content243 content307 - content-portion
    tray381 tray394 - tray
    table264 table339 table307 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray381 kitchen)
     (at tray394 kitchen)
     (at_kitchen_bread bread122)
     (at_kitchen_bread bread237)
     (at_kitchen_content content243)
     (at_kitchen_content content307)
     (no_gluten_bread bread237)
     (no_gluten_content content307)
     (allergic_gluten child254)
     (not_allergic_gluten child409)
     (waiting child254 table339)
     (waiting child409 table307)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child254)
     (served child409)
    )
  )
)
