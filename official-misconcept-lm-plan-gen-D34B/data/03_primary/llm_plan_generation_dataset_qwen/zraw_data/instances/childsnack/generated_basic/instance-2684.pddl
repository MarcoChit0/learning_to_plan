; child-snack task with 3 children and 0.23 gluten factor 
; constant factor of 1.9
; random seed: 866012

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child16 child93 child309 - child
    bread481 bread462 bread400 - bread-portion
    content286 content362 content249 - content-portion
    tray12 tray96 tray145 - tray
    table254 table255 table309 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray12 kitchen)
     (at tray96 kitchen)
     (at tray145 kitchen)
     (at_kitchen_bread bread481)
     (at_kitchen_bread bread462)
     (at_kitchen_bread bread400)
     (at_kitchen_content content286)
     (at_kitchen_content content362)
     (at_kitchen_content content249)
     (not_allergic_gluten child16)
     (not_allergic_gluten child309)
     (not_allergic_gluten child93)
     (waiting child16 table255)
     (waiting child93 table254)
     (waiting child309 table255)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child16)
     (served child93)
     (served child309)
    )
  )
)
