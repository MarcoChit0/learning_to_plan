; child-snack task with 2 children and 0.66 gluten factor 
; constant factor of 1.3
; random seed: 175785

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child462 child169 - child
    bread430 bread213 - bread-portion
    content77 content462 - content-portion
    tray374 tray72 tray436 - tray
    table376 table170 table200 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray374 kitchen)
     (at tray72 kitchen)
     (at tray436 kitchen)
     (at_kitchen_bread bread430)
     (at_kitchen_bread bread213)
     (at_kitchen_content content77)
     (at_kitchen_content content462)
     (no_gluten_bread bread430)
     (no_gluten_content content462)
     (allergic_gluten child169)
     (not_allergic_gluten child462)
     (waiting child462 table200)
     (waiting child169 table376)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child462)
     (served child169)
    )
  )
)
