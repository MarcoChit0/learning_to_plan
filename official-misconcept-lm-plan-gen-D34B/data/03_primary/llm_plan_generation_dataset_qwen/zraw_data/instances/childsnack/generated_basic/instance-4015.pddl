; child-snack task with 3 children and 0.22 gluten factor 
; constant factor of 1.9
; random seed: 386415

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child468 child375 child425 - child
    bread465 bread66 bread349 - bread-portion
    content85 content91 content321 - content-portion
    tray318 - tray
    table442 table254 table434 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray318 kitchen)
     (at_kitchen_bread bread465)
     (at_kitchen_bread bread66)
     (at_kitchen_bread bread349)
     (at_kitchen_content content85)
     (at_kitchen_content content91)
     (at_kitchen_content content321)
     (not_allergic_gluten child468)
     (not_allergic_gluten child375)
     (not_allergic_gluten child425)
     (waiting child468 table254)
     (waiting child375 table254)
     (waiting child425 table434)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child468)
     (served child375)
     (served child425)
    )
  )
)
