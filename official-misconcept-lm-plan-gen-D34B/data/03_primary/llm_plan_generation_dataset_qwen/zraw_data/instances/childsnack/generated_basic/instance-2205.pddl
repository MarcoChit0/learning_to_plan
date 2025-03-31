; child-snack task with 1 children and 0.73 gluten factor 
; constant factor of 1.5
; random seed: 455143

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child10 - child
    bread312 - bread-portion
    content188 - content-portion
    tray443 - tray
    table335 table52 table315 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray443 kitchen)
     (at_kitchen_bread bread312)
     (at_kitchen_content content188)
     (not_allergic_gluten child10)
     (waiting child10 table335)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child10)
    )
  )
)
