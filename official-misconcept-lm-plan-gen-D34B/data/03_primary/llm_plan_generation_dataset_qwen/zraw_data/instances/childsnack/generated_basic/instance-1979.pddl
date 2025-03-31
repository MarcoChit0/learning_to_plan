; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 81357

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child236 child254 - child
    bread487 bread348 - bread-portion
    content6 content362 - content-portion
    tray51 - tray
    table185 table182 table340 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray51 kitchen)
     (at_kitchen_bread bread487)
     (at_kitchen_bread bread348)
     (at_kitchen_content content6)
     (at_kitchen_content content362)
     (no_gluten_bread bread348)
     (no_gluten_content content6)
     (allergic_gluten child254)
     (not_allergic_gluten child236)
     (waiting child236 table340)
     (waiting child254 table185)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child236)
     (served child254)
    )
  )
)
