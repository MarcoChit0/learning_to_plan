; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 42775

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child328 child189 - child
    bread413 bread337 - bread-portion
    content98 content242 - content-portion
    tray444 tray150 - tray
    table327 table443 table56 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray444 kitchen)
     (at tray150 kitchen)
     (at_kitchen_bread bread413)
     (at_kitchen_bread bread337)
     (at_kitchen_content content98)
     (at_kitchen_content content242)
     (not_allergic_gluten child328)
     (not_allergic_gluten child189)
     (waiting child328 table56)
     (waiting child189 table56)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child328)
     (served child189)
    )
  )
)
