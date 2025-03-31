; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.9
; random seed: 394453

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child137 child358 - child
    bread377 bread345 - bread-portion
    content254 content74 - content-portion
    tray326 tray39 tray287 - tray
    table156 table457 table204 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray326 kitchen)
     (at tray39 kitchen)
     (at tray287 kitchen)
     (at_kitchen_bread bread377)
     (at_kitchen_bread bread345)
     (at_kitchen_content content254)
     (at_kitchen_content content74)
     (no_gluten_bread bread345)
     (no_gluten_content content74)
     (allergic_gluten child137)
     (not_allergic_gluten child358)
     (waiting child137 table156)
     (waiting child358 table156)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child137)
     (served child358)
    )
  )
)
