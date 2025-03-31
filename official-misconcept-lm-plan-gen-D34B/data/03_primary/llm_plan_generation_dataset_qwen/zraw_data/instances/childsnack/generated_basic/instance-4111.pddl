; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 104895

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child278 child201 - child
    bread10 bread134 - bread-portion
    content287 content213 - content-portion
    tray457 tray41 tray330 - tray
    table50 table168 table437 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray457 kitchen)
     (at tray41 kitchen)
     (at tray330 kitchen)
     (at_kitchen_bread bread10)
     (at_kitchen_bread bread134)
     (at_kitchen_content content287)
     (at_kitchen_content content213)
     (not_allergic_gluten child278)
     (not_allergic_gluten child201)
     (waiting child278 table50)
     (waiting child201 table437)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child278)
     (served child201)
    )
  )
)
