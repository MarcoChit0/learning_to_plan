; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 103060

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child112 child215 child434 - child
    bread233 bread43 bread307 - bread-portion
    content454 content87 content382 - content-portion
    tray72 - tray
    table254 table315 table60 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray72 kitchen)
     (at_kitchen_bread bread233)
     (at_kitchen_bread bread43)
     (at_kitchen_bread bread307)
     (at_kitchen_content content454)
     (at_kitchen_content content87)
     (at_kitchen_content content382)
     (not_allergic_gluten child215)
     (not_allergic_gluten child112)
     (not_allergic_gluten child434)
     (waiting child112 table254)
     (waiting child215 table315)
     (waiting child434 table60)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child112)
     (served child215)
     (served child434)
    )
  )
)
