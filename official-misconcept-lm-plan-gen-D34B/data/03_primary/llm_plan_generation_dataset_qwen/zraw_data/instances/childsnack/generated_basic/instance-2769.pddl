; child-snack task with 2 children and 0.77 gluten factor 
; constant factor of 1.3
; random seed: 672548

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child220 child210 - child
    bread489 bread64 - bread-portion
    content395 content137 - content-portion
    tray403 - tray
    table127 table123 table358 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray403 kitchen)
     (at_kitchen_bread bread489)
     (at_kitchen_bread bread64)
     (at_kitchen_content content395)
     (at_kitchen_content content137)
     (no_gluten_bread bread489)
     (no_gluten_content content395)
     (allergic_gluten child210)
     (not_allergic_gluten child220)
     (waiting child220 table127)
     (waiting child210 table127)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child220)
     (served child210)
    )
  )
)
