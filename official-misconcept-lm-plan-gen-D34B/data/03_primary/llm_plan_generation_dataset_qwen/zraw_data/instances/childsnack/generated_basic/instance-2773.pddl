; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 484368

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child181 child35 - child
    bread201 bread366 - bread-portion
    content495 content330 - content-portion
    tray95 - tray
    table226 table480 table446 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray95 kitchen)
     (at_kitchen_bread bread201)
     (at_kitchen_bread bread366)
     (at_kitchen_content content495)
     (at_kitchen_content content330)
     (no_gluten_bread bread366)
     (no_gluten_content content495)
     (allergic_gluten child181)
     (not_allergic_gluten child35)
     (waiting child181 table226)
     (waiting child35 table226)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child181)
     (served child35)
    )
  )
)
