; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 233159

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child336 child93 - child
    bread85 bread120 - bread-portion
    content409 content353 - content-portion
    tray262 - tray
    table189 table441 table183 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray262 kitchen)
     (at_kitchen_bread bread85)
     (at_kitchen_bread bread120)
     (at_kitchen_content content409)
     (at_kitchen_content content353)
     (no_gluten_bread bread120)
     (no_gluten_content content409)
     (allergic_gluten child93)
     (not_allergic_gluten child336)
     (waiting child336 table189)
     (waiting child93 table189)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child336)
     (served child93)
    )
  )
)
