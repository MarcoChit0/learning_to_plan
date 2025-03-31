; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.9
; random seed: 206706

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child227 child271 - child
    bread208 bread0 - bread-portion
    content364 content457 - content-portion
    tray308 - tray
    table248 table169 table333 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray308 kitchen)
     (at_kitchen_bread bread208)
     (at_kitchen_bread bread0)
     (at_kitchen_content content364)
     (at_kitchen_content content457)
     (no_gluten_bread bread208)
     (no_gluten_content content364)
     (allergic_gluten child271)
     (not_allergic_gluten child227)
     (waiting child227 table333)
     (waiting child271 table248)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child227)
     (served child271)
    )
  )
)
