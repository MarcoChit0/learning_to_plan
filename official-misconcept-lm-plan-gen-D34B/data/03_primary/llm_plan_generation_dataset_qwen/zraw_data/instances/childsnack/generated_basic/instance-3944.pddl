; child-snack task with 2 children and 0.78 gluten factor 
; constant factor of 1.9
; random seed: 272364

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child336 child329 - child
    bread449 bread499 - bread-portion
    content393 content4 - content-portion
    tray98 - tray
    table257 table158 table207 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray98 kitchen)
     (at_kitchen_bread bread449)
     (at_kitchen_bread bread499)
     (at_kitchen_content content393)
     (at_kitchen_content content4)
     (no_gluten_bread bread449)
     (no_gluten_content content4)
     (allergic_gluten child336)
     (not_allergic_gluten child329)
     (waiting child336 table207)
     (waiting child329 table158)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child336)
     (served child329)
    )
  )
)
