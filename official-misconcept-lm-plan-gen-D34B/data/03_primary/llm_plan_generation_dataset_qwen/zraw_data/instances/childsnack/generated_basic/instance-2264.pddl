; child-snack task with 2 children and 0.54 gluten factor 
; constant factor of 1.7
; random seed: 118280

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child123 child152 - child
    bread430 bread140 - bread-portion
    content75 content343 - content-portion
    tray164 - tray
    table247 table223 table193 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray164 kitchen)
     (at_kitchen_bread bread430)
     (at_kitchen_bread bread140)
     (at_kitchen_content content75)
     (at_kitchen_content content343)
     (no_gluten_bread bread140)
     (no_gluten_content content343)
     (allergic_gluten child152)
     (not_allergic_gluten child123)
     (waiting child123 table193)
     (waiting child152 table193)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child123)
     (served child152)
    )
  )
)
