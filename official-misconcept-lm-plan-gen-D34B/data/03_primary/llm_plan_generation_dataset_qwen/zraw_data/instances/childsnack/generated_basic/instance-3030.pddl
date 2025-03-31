; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 872166

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child68 child152 - child
    bread181 bread62 - bread-portion
    content381 content229 - content-portion
    tray16 tray15 tray80 - tray
    table8 table336 table331 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray16 kitchen)
     (at tray15 kitchen)
     (at tray80 kitchen)
     (at_kitchen_bread bread181)
     (at_kitchen_bread bread62)
     (at_kitchen_content content381)
     (at_kitchen_content content229)
     (no_gluten_bread bread181)
     (no_gluten_content content381)
     (allergic_gluten child152)
     (not_allergic_gluten child68)
     (waiting child68 table336)
     (waiting child152 table336)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child68)
     (served child152)
    )
  )
)
