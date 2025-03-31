; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 924791

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child306 - child
    bread66 - bread-portion
    content363 - content-portion
    tray67 tray200 tray358 - tray
    table18 table141 table470 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray67 kitchen)
     (at tray200 kitchen)
     (at tray358 kitchen)
     (at_kitchen_bread bread66)
     (at_kitchen_content content363)
     (not_allergic_gluten child306)
     (waiting child306 table141)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child306)
    )
  )
)
