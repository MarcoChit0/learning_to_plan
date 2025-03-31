; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 703301

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child3 child199 - child
    bread418 bread393 - bread-portion
    content410 content157 - content-portion
    tray483 - tray
    table209 table185 table24 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray483 kitchen)
     (at_kitchen_bread bread418)
     (at_kitchen_bread bread393)
     (at_kitchen_content content410)
     (at_kitchen_content content157)
     (no_gluten_bread bread418)
     (no_gluten_content content410)
     (allergic_gluten child3)
     (not_allergic_gluten child199)
     (waiting child3 table209)
     (waiting child199 table24)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child3)
     (served child199)
    )
  )
)
