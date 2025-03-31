; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 908360

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child68 - child
    bread485 - bread-portion
    content146 - content-portion
    tray365 tray358 tray331 tray19 - tray
    table336 table315 table346 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray365 kitchen)
     (at tray358 kitchen)
     (at tray331 kitchen)
     (at tray19 kitchen)
     (at_kitchen_bread bread485)
     (at_kitchen_content content146)
     (not_allergic_gluten child68)
     (waiting child68 table346)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child68)
    )
  )
)
