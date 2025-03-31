; child-snack task with 2 children and 0.61 gluten factor 
; constant factor of 1.3
; random seed: 971660

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child171 child463 - child
    bread362 bread336 - bread-portion
    content372 content277 - content-portion
    tray456 tray87 tray156 tray91 - tray
    table498 table80 table9 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray456 kitchen)
     (at tray87 kitchen)
     (at tray156 kitchen)
     (at tray91 kitchen)
     (at_kitchen_bread bread362)
     (at_kitchen_bread bread336)
     (at_kitchen_content content372)
     (at_kitchen_content content277)
     (no_gluten_bread bread336)
     (no_gluten_content content277)
     (allergic_gluten child463)
     (not_allergic_gluten child171)
     (waiting child171 table80)
     (waiting child463 table498)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child171)
     (served child463)
    )
  )
)
