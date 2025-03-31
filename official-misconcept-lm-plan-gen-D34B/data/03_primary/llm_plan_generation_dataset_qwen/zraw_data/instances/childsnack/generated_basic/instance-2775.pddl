; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 127020

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child363 child400 - child
    bread443 bread138 - bread-portion
    content186 content30 - content-portion
    tray134 - tray
    table87 table472 table317 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray134 kitchen)
     (at_kitchen_bread bread443)
     (at_kitchen_bread bread138)
     (at_kitchen_content content186)
     (at_kitchen_content content30)
     (no_gluten_bread bread138)
     (no_gluten_content content30)
     (allergic_gluten child363)
     (not_allergic_gluten child400)
     (waiting child363 table317)
     (waiting child400 table87)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child363)
     (served child400)
    )
  )
)
