; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 191808

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child64 child251 - child
    bread160 bread335 - bread-portion
    content65 content132 - content-portion
    tray297 - tray
    table193 table271 table292 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray297 kitchen)
     (at_kitchen_bread bread160)
     (at_kitchen_bread bread335)
     (at_kitchen_content content65)
     (at_kitchen_content content132)
     (no_gluten_bread bread160)
     (no_gluten_content content132)
     (allergic_gluten child251)
     (not_allergic_gluten child64)
     (waiting child64 table271)
     (waiting child251 table271)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child64)
     (served child251)
    )
  )
)
