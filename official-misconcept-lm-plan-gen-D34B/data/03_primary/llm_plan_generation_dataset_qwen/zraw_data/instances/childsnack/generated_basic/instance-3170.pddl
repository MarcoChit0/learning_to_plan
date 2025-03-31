; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 489009

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child292 child323 - child
    bread468 bread255 - bread-portion
    content119 content198 - content-portion
    tray53 tray434 - tray
    table489 table462 table299 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray53 kitchen)
     (at tray434 kitchen)
     (at_kitchen_bread bread468)
     (at_kitchen_bread bread255)
     (at_kitchen_content content119)
     (at_kitchen_content content198)
     (not_allergic_gluten child292)
     (not_allergic_gluten child323)
     (waiting child292 table462)
     (waiting child323 table299)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child292)
     (served child323)
    )
  )
)
