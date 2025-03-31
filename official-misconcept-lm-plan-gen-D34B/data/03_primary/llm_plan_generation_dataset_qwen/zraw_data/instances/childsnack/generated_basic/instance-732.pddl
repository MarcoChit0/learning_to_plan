; child-snack task with 2 children and 0.64 gluten factor 
; constant factor of 1.9
; random seed: 864631

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child479 child381 - child
    bread278 bread28 - bread-portion
    content363 content202 - content-portion
    tray248 tray128 - tray
    table223 table56 table285 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray248 kitchen)
     (at tray128 kitchen)
     (at_kitchen_bread bread278)
     (at_kitchen_bread bread28)
     (at_kitchen_content content363)
     (at_kitchen_content content202)
     (no_gluten_bread bread278)
     (no_gluten_content content202)
     (allergic_gluten child381)
     (not_allergic_gluten child479)
     (waiting child479 table285)
     (waiting child381 table56)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child479)
     (served child381)
    )
  )
)
