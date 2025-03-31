; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 341250

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child254 child13 - child
    bread99 bread150 - bread-portion
    content266 content370 - content-portion
    tray126 - tray
    table497 table91 table98 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray126 kitchen)
     (at_kitchen_bread bread99)
     (at_kitchen_bread bread150)
     (at_kitchen_content content266)
     (at_kitchen_content content370)
     (no_gluten_bread bread99)
     (no_gluten_content content266)
     (allergic_gluten child13)
     (not_allergic_gluten child254)
     (waiting child254 table91)
     (waiting child13 table98)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child254)
     (served child13)
    )
  )
)
