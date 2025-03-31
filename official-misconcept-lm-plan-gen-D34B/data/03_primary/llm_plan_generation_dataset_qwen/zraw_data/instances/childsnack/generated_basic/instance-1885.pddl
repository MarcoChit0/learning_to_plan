; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.7
; random seed: 549851

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child481 child175 - child
    bread314 bread201 - bread-portion
    content435 content316 - content-portion
    tray371 tray300 - tray
    table407 table63 table182 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray371 kitchen)
     (at tray300 kitchen)
     (at_kitchen_bread bread314)
     (at_kitchen_bread bread201)
     (at_kitchen_content content435)
     (at_kitchen_content content316)
     (no_gluten_bread bread314)
     (no_gluten_content content316)
     (allergic_gluten child481)
     (not_allergic_gluten child175)
     (waiting child481 table63)
     (waiting child175 table63)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child481)
     (served child175)
    )
  )
)
