; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.3
; random seed: 152276

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child33 child465 - child
    bread158 bread254 - bread-portion
    content188 content494 - content-portion
    tray226 - tray
    table339 table475 table64 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray226 kitchen)
     (at_kitchen_bread bread158)
     (at_kitchen_bread bread254)
     (at_kitchen_content content188)
     (at_kitchen_content content494)
     (no_gluten_bread bread254)
     (no_gluten_content content188)
     (allergic_gluten child465)
     (not_allergic_gluten child33)
     (waiting child33 table64)
     (waiting child465 table339)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child33)
     (served child465)
    )
  )
)
