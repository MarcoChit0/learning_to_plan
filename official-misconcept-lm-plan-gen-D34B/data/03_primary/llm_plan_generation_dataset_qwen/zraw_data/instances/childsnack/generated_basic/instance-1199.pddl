; child-snack task with 3 children and 0.33 gluten factor 
; constant factor of 1.7
; random seed: 776424

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child363 child350 child82 - child
    bread178 bread85 bread416 - bread-portion
    content37 content224 content67 - content-portion
    tray180 tray336 tray248 - tray
    table224 table366 table409 - place
    sandw1 sandw2 sandw3 sandw4 sandw5 sandw6 - sandwich
  )
  (:init
     (at tray180 kitchen)
     (at tray336 kitchen)
     (at tray248 kitchen)
     (at_kitchen_bread bread178)
     (at_kitchen_bread bread85)
     (at_kitchen_bread bread416)
     (at_kitchen_content content37)
     (at_kitchen_content content224)
     (at_kitchen_content content67)
     (not_allergic_gluten child82)
     (not_allergic_gluten child363)
     (not_allergic_gluten child350)
     (waiting child363 table224)
     (waiting child350 table224)
     (waiting child82 table366)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
     (notexist sandw5)
     (notexist sandw6)
  )
  (:goal
    (and
     (served child363)
     (served child350)
     (served child82)
    )
  )
)
