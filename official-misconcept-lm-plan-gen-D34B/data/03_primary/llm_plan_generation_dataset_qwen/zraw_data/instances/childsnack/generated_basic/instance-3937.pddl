; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 336869

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child62 child243 - child
    bread178 bread415 - bread-portion
    content379 content279 - content-portion
    tray335 - tray
    table33 table225 table446 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray335 kitchen)
     (at_kitchen_bread bread178)
     (at_kitchen_bread bread415)
     (at_kitchen_content content379)
     (at_kitchen_content content279)
     (no_gluten_bread bread415)
     (no_gluten_content content279)
     (allergic_gluten child243)
     (not_allergic_gluten child62)
     (waiting child62 table225)
     (waiting child243 table33)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child62)
     (served child243)
    )
  )
)
