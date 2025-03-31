; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.9
; random seed: 510776

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child41 child156 - child
    bread227 bread89 - bread-portion
    content64 content406 - content-portion
    tray355 tray381 tray128 tray14 - tray
    table204 table402 table253 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray355 kitchen)
     (at tray381 kitchen)
     (at tray128 kitchen)
     (at tray14 kitchen)
     (at_kitchen_bread bread227)
     (at_kitchen_bread bread89)
     (at_kitchen_content content64)
     (at_kitchen_content content406)
     (no_gluten_bread bread89)
     (no_gluten_content content406)
     (allergic_gluten child41)
     (not_allergic_gluten child156)
     (waiting child41 table204)
     (waiting child156 table253)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child41)
     (served child156)
    )
  )
)
