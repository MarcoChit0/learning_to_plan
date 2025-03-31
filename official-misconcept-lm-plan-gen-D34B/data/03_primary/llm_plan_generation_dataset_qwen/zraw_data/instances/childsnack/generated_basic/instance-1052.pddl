; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 871408

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child381 child311 - child
    bread492 bread413 - bread-portion
    content413 content178 - content-portion
    tray465 tray105 tray375 - tray
    table241 table346 table334 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray465 kitchen)
     (at tray105 kitchen)
     (at tray375 kitchen)
     (at_kitchen_bread bread492)
     (at_kitchen_bread bread413)
     (at_kitchen_content content413)
     (at_kitchen_content content178)
     (not_allergic_gluten child381)
     (not_allergic_gluten child311)
     (waiting child381 table334)
     (waiting child311 table346)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child381)
     (served child311)
    )
  )
)
