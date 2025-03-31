; child-snack task with 2 children and 0.31 gluten factor 
; constant factor of 1.7
; random seed: 713196

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child229 child299 - child
    bread220 bread399 - bread-portion
    content303 content350 - content-portion
    tray221 tray342 tray389 - tray
    table45 table122 table53 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray221 kitchen)
     (at tray342 kitchen)
     (at tray389 kitchen)
     (at_kitchen_bread bread220)
     (at_kitchen_bread bread399)
     (at_kitchen_content content303)
     (at_kitchen_content content350)
     (not_allergic_gluten child229)
     (not_allergic_gluten child299)
     (waiting child229 table53)
     (waiting child299 table53)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child229)
     (served child299)
    )
  )
)
