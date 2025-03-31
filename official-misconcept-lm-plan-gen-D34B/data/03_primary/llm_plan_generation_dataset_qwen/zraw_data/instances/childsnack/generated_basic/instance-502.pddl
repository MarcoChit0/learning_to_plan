; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.9
; random seed: 637727

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child252 child78 - child
    bread228 bread206 - bread-portion
    content275 content80 - content-portion
    tray386 tray4 - tray
    table60 table316 table281 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray386 kitchen)
     (at tray4 kitchen)
     (at_kitchen_bread bread228)
     (at_kitchen_bread bread206)
     (at_kitchen_content content275)
     (at_kitchen_content content80)
     (not_allergic_gluten child78)
     (not_allergic_gluten child252)
     (waiting child252 table281)
     (waiting child78 table281)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child252)
     (served child78)
    )
  )
)
