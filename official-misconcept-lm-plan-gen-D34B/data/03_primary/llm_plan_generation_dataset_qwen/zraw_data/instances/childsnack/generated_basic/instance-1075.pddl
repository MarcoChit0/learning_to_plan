; child-snack task with 2 children and 0.41 gluten factor 
; constant factor of 1.3
; random seed: 518299

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child368 child115 - child
    bread74 bread58 - bread-portion
    content418 content79 - content-portion
    tray138 tray485 tray177 - tray
    table167 table300 table499 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray138 kitchen)
     (at tray485 kitchen)
     (at tray177 kitchen)
     (at_kitchen_bread bread74)
     (at_kitchen_bread bread58)
     (at_kitchen_content content418)
     (at_kitchen_content content79)
     (not_allergic_gluten child368)
     (not_allergic_gluten child115)
     (waiting child368 table499)
     (waiting child115 table499)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child368)
     (served child115)
    )
  )
)
