; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 440165

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child9 child399 - child
    bread24 bread217 - bread-portion
    content282 content271 - content-portion
    tray95 tray297 tray348 - tray
    table446 table269 table345 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray95 kitchen)
     (at tray297 kitchen)
     (at tray348 kitchen)
     (at_kitchen_bread bread24)
     (at_kitchen_bread bread217)
     (at_kitchen_content content282)
     (at_kitchen_content content271)
     (not_allergic_gluten child9)
     (not_allergic_gluten child399)
     (waiting child9 table269)
     (waiting child399 table446)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child9)
     (served child399)
    )
  )
)
