; child-snack task with 1 children and 0.45 gluten factor 
; constant factor of 1.9
; random seed: 738619

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child419 - child
    bread234 - bread-portion
    content134 - content-portion
    tray259 tray330 tray63 - tray
    table214 table261 table315 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray259 kitchen)
     (at tray330 kitchen)
     (at tray63 kitchen)
     (at_kitchen_bread bread234)
     (at_kitchen_content content134)
     (not_allergic_gluten child419)
     (waiting child419 table261)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child419)
    )
  )
)
