; child-snack task with 2 children and 0.27 gluten factor 
; constant factor of 1.7
; random seed: 60025

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child120 child164 - child
    bread192 bread254 - bread-portion
    content307 content230 - content-portion
    tray277 tray136 tray149 - tray
    table372 table307 table156 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray277 kitchen)
     (at tray136 kitchen)
     (at tray149 kitchen)
     (at_kitchen_bread bread192)
     (at_kitchen_bread bread254)
     (at_kitchen_content content307)
     (at_kitchen_content content230)
     (not_allergic_gluten child120)
     (not_allergic_gluten child164)
     (waiting child120 table372)
     (waiting child164 table372)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child120)
     (served child164)
    )
  )
)
