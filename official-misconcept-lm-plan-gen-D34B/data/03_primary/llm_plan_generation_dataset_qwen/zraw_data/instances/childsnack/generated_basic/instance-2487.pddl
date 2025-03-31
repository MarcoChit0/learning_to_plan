; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 336333

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child267 child128 - child
    bread258 bread436 - bread-portion
    content188 content377 - content-portion
    tray491 tray93 - tray
    table10 table204 table39 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray491 kitchen)
     (at tray93 kitchen)
     (at_kitchen_bread bread258)
     (at_kitchen_bread bread436)
     (at_kitchen_content content188)
     (at_kitchen_content content377)
     (not_allergic_gluten child128)
     (not_allergic_gluten child267)
     (waiting child267 table204)
     (waiting child128 table204)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child267)
     (served child128)
    )
  )
)
